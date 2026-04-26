/* BEGIN FUNCTION-DEF rgba_desaturate LOC=UNKNOWN VKEY=4891 */
void rgba_desaturate(struct rgba *dest , struct rgba *src ) 
{ 
  double avg ;
  unsigned long _TIG_FN_FQqO_1_rgba_desaturate_next ;

  {
  _TIG_FN_FQqO_1_rgba_desaturate_next = 0UL;
  while (1) {
    switch (_TIG_FN_FQqO_1_rgba_desaturate_next) {
    case 1UL: 
    {
#line 43
    avg = (double )(((src->r + src->g) + src->b) / 3);
#line 45
    dest->r = (int )avg;
#line 46
    dest->g = (int )avg;
#line 47
    dest->b = (int )avg;
#line 48
    dest->a = src->a;
    }
    _TIG_FN_FQqO_1_rgba_desaturate_next = 2UL;
    break;
    case 0UL: ;
    _TIG_FN_FQqO_1_rgba_desaturate_next = 1UL;
    break;
    case 2UL: ;
    return;
    break;
    default: 
    break;
    }
  }
}
}
/* END FUNCTION-DEF rgba_desaturate LOC=UNKNOWN VKEY=4891 */